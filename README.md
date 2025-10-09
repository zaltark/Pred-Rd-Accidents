# Kaggle Playground Series - Predicting Road Accident Risk

## Overview
Welcome to the 2025 Kaggle Playground Series! We plan to continue in the spirit of previous playgrounds, providing interesting and approachable datasets for our community to practice their machine learning skills, and anticipate a competition each month.

**Your Goal:** Predict the likelihood of accidents on different types of roads.

For this Playground Series challenge, we have teamed up with Stack Overflow to give you a two-part challenge. The Stack Overflow Challenge is the second part and builds upon this one by having participants develop a web application. We encourage you to check out the Stack Overflow Challenge!

If you complete both challenges, we’ll recognize your breadth of skills with a special “Code Scientist” badge which will appear on both Kaggle and Stack Overflow.

## Timeline
- **Start Date:** October 1, 2025
- **Entry Deadline:** Same as the Final Submission Deadline
- **Team Merger Deadline:** Same as the Final Submission Deadline
- **Final Submission Deadline:** October 31, 2025

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

## Evaluation
Submissions are evaluated using the Root Mean Squared Error between the predicted and the observed target.

## Submission File
For each `id` in the test set, you must predict a `accident_risk` of between 0 and 1. The file should contain a header and have the following format:

```
id,accident_risk
517754,0.352
517755,0.992
517756,0.021
etc.
```

## About the Tabular Playground Series
The goal of the Tabular Playground Series is to provide the Kaggle community with a variety of fairly light-weight challenges that can be used to learn and sharpen skills in different aspects of machine learning and data science. The duration of each competition will generally only last a few weeks, and may have longer or shorter durations depending on the challenge. The challenges will generally use fairly light-weight datasets that are synthetically generated from real-world data, and will provide an opportunity to quickly iterate through various model and feature engineering ideas, create visualizations, etc.

### Synthetically-Generated Datasets
Using synthetic data for Playground competitions allows us to strike a balance between having real-world data (with named features) and ensuring test labels are not publicly available. This allows us to host competitions with more interesting datasets than in the past. While there are still challenges with synthetic data generation, the state-of-the-art is much better now than when we started the Tabular Playground Series two years ago, and that goal is to produce datasets that have far fewer artifacts. Please feel free to give us feedback on the datasets for the different competitions so that we can continue to improve!

## Prizes
- **1st Place:** Choice of Kaggle merchandise
- **2nd Place:** Choice of Kaggle merchandise
- **3rd Place:** Choice of Kaggle merchandise

Please note: In order to encourage more participation from beginners, Kaggle merchandise will only be awarded once per person in this series. If a person has previously won, we'll skip to the next team.

## Session Summary: October 8, 2025 - Deep Dive & Model Refinement

This session focused on an extensive deep dive into the dataset, iterative refinement of our feature engineering strategy, and initial model development and tuning.

### Key Achievements & Findings:

*   **Comprehensive EDA & Visualization**: Conducted thorough exploratory data analysis across all features, including detailed deep dives into environmental, temporal, and road characteristics. All visualization scripts were made robust and reports updated to reflect the latest insights.
*   **Refined Feature Engineering Plan**: Developed a highly detailed feature engineering plan, documented in `data/README.md`, which includes:
    *   Clarifying `lighting` categories to `Bright`, `Dim`, `Dark`.
    *   Creating numerous interaction and leveled features: `Visibility_Score`, `school_season_x_time_of_day`, `high_risk_curvature_interaction`, `speed_zone`, `is_unsigned_urban_road`, `is_accident_hotspot`, `lane_category`, `is_urban_evening_clear_weather`, and `accident_hotspot_x_weather`.
*   **Feature Engineering Pipeline Implementation**: Successfully implemented the `build_features.py` script to transform raw data into our final engineered feature set.
*   **Baseline Model Testing**: Established baseline performance for 5 regression models (XGBoost, LightGBM, RandomForest, CatBoost, Ridge) using the full engineered feature set.
*   **Hyperparameter Tuning (XGBoost & CatBoost)**: Utilized Optuna to tune XGBoost and CatBoost, achieving improved RMSE scores.
*   **Residual Analysis & Stacking Model**: Investigated residual patterns, identifying heteroscedasticity. Implemented a stacking model (XGBoost base + tuned XGBoost residual) to address these systematic errors.
*   **Target Transformation Experiment**: Explored logit transformation of the target variable to address heteroscedasticity, but this approach resulted in a worse Public Score and was reverted.
*   **Best Model Performance**:
    *   **Best Validation RMSE**: 0.076668 (achieved by the Stacking model with tuned residual component).
    *   **Best Public Score**: 0.07576 (achieved by the Stacking model without target transformation).

### Next Steps:

*   Further investigate the residual patterns of our best model to identify new feature engineering opportunities.
*   Explore more advanced ensemble techniques or alternative model architectures.
*   Consider more extensive hyperparameter tuning or broader search spaces.
