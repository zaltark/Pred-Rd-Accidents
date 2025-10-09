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

3.  **`holiday`**: The mean `accident_risk` is consistently higher on holidays compared to non-holidays, across all times of day. This suggests that `holiday` is a strong predictive feature.

4.  **`school_season` and `time_of_day` Interaction**: These two features have a complex interaction. Accident risk is higher in the afternoon during the school season (likely due to school letting out), but surprisingly *lower* in the morning. This indicates that a simple boolean `school_season` feature is not enough to capture the full pattern.

## Proposed Feature Engineering

Based on our EDA, the following feature engineering steps are proposed before modeling:

1.  **Clarify `lighting` Categories**: As a first step, we will rename the categories in the `lighting` column to our new, more descriptive luminosity ratings.
    *   `daylight` → **`Bright`**
    *   `dim` → **`Dim`**
    *   `night` → **`Dark`**

2.  **Create Interaction & Leveled Features**: Next, we will engineer new features to capture the complex, non-linear relationships discovered during our analysis.
    *   **Visibility Score**: Using our new `lighting` categories, we will create a leveled, numerical feature to capture the combined risk of luminosity and `weather`. For example:
        - **Level 0 (Best)**: Bright & Clear
        - **Level 1 (Moderate)**: Bright with Rain/Fog, OR Dim & Clear
        - **Level 2 (Poor)**: Dark & Clear, OR Dim with Rain/Fog
        - **Level 3 (Worst)**: Dark with Rain/Fog
    *   **`school_season_x_time_of_day`**: An interaction feature to capture the unique risk patterns of school season mornings and afternoons.
    *   **`holiday_x_time_of_day_x_bright`**: A binary feature to flag the specific high-risk combination of a holiday, a specific time of day (e.g., afternoon/evening), and bright lighting. This is to help the model identify risks that are not weather-related but time and holiday dependent.
    *   **`high_risk_curvature_interaction`**: A feature that multiplies `curvature` by a flag for high-risk conditions (e.g., `speed_limit` > 55 or `road_type` is 'highway') to capture the added danger of curves on fast roads.
    *   **`speed_zone`**: A categorical feature to capture the non-linear effect of speed limit on risk. This will be created by binning `speed_limit` into two groups: `'low_speed'` (<= 45 mph) and `'high_speed'` (> 45 mph).
    *   **`is_unsigned_urban_road`**: A binary feature that flags the specific, high-risk combination of a road being both 'urban' and having no road signs present.
    *   **`is_accident_hotspot`**: A binary feature to flag roads that are known accident hotspots. This will be `True` if `num_reported_accidents` is 3 or more, and `False` otherwise.
    *   **`lane_category`**: A categorical feature derived from `num_lanes` to capture non-linear risk patterns: `'single_lane'` (1), `'two_lanes'` (2), `'three_lanes'` (3), and `'multi_lanes'` (>= 4).
    *   **`is_urban_evening_clear_weather`**: A binary feature flagging the specific high-risk combination of 'urban' `road_type`, 'evening' `time_of_day`, and 'clear' `weather`.
    *   **`accident_hotspot_x_weather`**: A categorical interaction feature combining the `is_accident_hotspot` flag with `weather` conditions, to capture the weather-dependent threshold for accident hotspots.
    *   **`low_speed_x_road_type_urban`**: A binary interaction feature to identify urban roads within low-speed zones.
    *   **`low_speed_x_lane_category_two_lanes`**: A binary interaction feature to identify two-lane roads within low-speed zones.
    *   **`low_speed_x_Visibility_Score`**: A numerical interaction feature to capture the combined effect of low-speed zones and visibility.
    *   **`is_moderate_visibility_overprediction_zone`**: A binary feature to flag instances where `Visibility_Score` is 1 or 2, indicating a zone where the model systematically overpredicts risk.

3.  **Final Encoding**:
    *   The new `Visibility Score` and other engineered numerical features will be scaled.
    *   The remaining categorical features (like `road_type`, `weather`, etc.) will be one-hot encoded.
