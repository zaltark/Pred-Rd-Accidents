# Data Management Plan

This document outlines the strategy for managing data in this project.

## Data Pipeline

The data pipeline consists of three stages, corresponding to the subdirectories in this folder:

1.  **Raw Data (`data/raw`)**: This directory is the starting point for our data pipeline. It should contain the original, immutable data files. For this project, this includes `train.csv` and `test.csv` from the Kaggle competition.

2.  **Processed Data (`data/processed`)**: This directory will store the cleaned, transformed, and feature-engineered data. The scripts in `src/data` and `src/features` will take the raw data as input and produce processed data in this directory. This may include:
    *   Handling missing values.
    *   Encoding categorical features.
    *   Scaling numerical features.
    *   Creating new features (feature engineering).

3.  **External Data (`data/external`)**: This directory is for any additional data sources that may be used to enrich the original dataset. For now, this directory is a placeholder.

## Data Versioning

While not yet implemented, we should consider using a data versioning tool like [DVC (Data Version Control)](https://dvc.org/) in the future. This will allow us to track changes in our data and models, ensuring reproducibility.

## Data Summary (Initial EDA)

Based on our initial Exploratory Data Analysis, here is a summary of the training data:

*   **Size**: The dataset contains 517,754 rows and 14 columns.
*   **Missing Values**: There are no missing values in the dataset.
*   **Column Types**:
    *   **Numeric**: `id`, `num_lanes`, `curvature`, `speed_limit`, `num_reported_accidents`, `accident_risk` (target variable).
    *   **Categorical**: `road_type`, `lighting`, `weather`, `time_of_day`.
    *   **Boolean**: `road_signs_present`, `public_road`, `holiday`, `school_season`.
*   **Target Variable**: `accident_risk` is a float between 0 and 1.

## Key Findings from Deeper Analysis

1.  **`road_signs_present`**: Our deep-dive analysis shows that this feature has a negligible effect on the *median* accident risk, even when controlling for `road_type`. While a model might find a subtle signal, this is not a strong predictor on its own.

2.  **`road_type`**: The 'urban' road type is associated with a noticeably higher mean `accident_risk` compared to 'rural' and 'highway'. This feature is likely acting as a proxy for latent variables such as traffic density, intersection complexity, and pedestrian activity.

## Proposed Feature Engineering

Based on our EDA, the following feature engineering steps are proposed before modeling:

1.  **Encode Categorical Variables**: Machine learning models require numerical input. All categorical features (`road_type`, `lighting`, `weather`, `time_of_day`, and boolean features) will need to be converted to a numerical format.

2.  **One-Hot Encoding**: This is the proposed method for encoding our categorical features. It will create new binary columns for each category (e.g., `is_urban`, `is_rural`), allowing the model to learn a specific weight for each one.

3.  **Create Interaction Features**: We should explore creating new features by combining existing ones. For example, a 'poor_visibility' feature could be created by combining specific categories from the `lighting` and `weather` columns. This could capture non-linear relationships and provide more predictive power.
