
import pandas as pd
import numpy as np
import os

def build_features(df):
    """
    Applies all defined feature engineering steps to the input DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing raw features.

    Returns:
        pd.DataFrame: The DataFrame with engineered features.
    """

    # 1. Clarify lighting Categories
    lighting_map = {
        'daylight': 'Bright',
        'dim': 'Dim',
        'night': 'Dark'
    }
    df['lighting'] = df['lighting'].replace(lighting_map)

    # 2. Create Interaction & Leveled Features

    # 2.1 Visibility Score
    weather_map = {
        'clear': 'Clear',
        'rainy': 'Rainy/Foggy', # Group rainy and foggy for simplicity in score
        'foggy': 'Rainy/Foggy'
    }
    df['weather_mapped'] = df['weather'].replace(weather_map) # Create a temporary mapped weather column

    def calculate_visibility_score(row):
        lighting = row['lighting']
        weather = row['weather_mapped'] # Use the mapped weather

        if lighting == 'Bright' and weather == 'Clear':
            return 0
        elif (lighting == 'Bright' and weather == 'Rainy/Foggy') or \
             (lighting == 'Dim' and weather == 'Clear'):
            return 1
        elif (lighting == 'Dark' and weather == 'Clear') or \
             (lighting == 'Dim' and weather == 'Rainy/Foggy'):
            return 2
        elif lighting == 'Dark' and weather == 'Rainy/Foggy':
            return 3
        else:
            return np.nan # Should not happen with our defined categories

    df['Visibility_Score'] = df.apply(calculate_visibility_score, axis=1)
    df = df.drop('weather_mapped', axis=1) # Drop the temporary column

    # 2.2 school_season_x_time_of_day
    df['school_season_x_time_of_day'] = df.apply(
        lambda row: f"{ 'School' if row['school_season'] else 'NonSchool'}_{row['time_of_day']}",
        axis=1
    )

    # 2.3 holiday_x_time_of_day_x_bright
    df['holiday_x_time_of_day_x_bright'] = (
        (df['holiday'] == True) &
        (df['time_of_day'].isin(['afternoon', 'evening'])) &
        (df['lighting'] == 'Bright')
    )

    # 2.4 high_risk_curvature_interaction
    # Define high-risk speed/road conditions
    df['is_high_risk_road_condition'] = ((df['speed_limit'] > 55) | (df['road_type'] == 'highway'))
    df['high_risk_curvature_interaction'] = df['curvature'] * df['is_high_risk_road_condition']
    df = df.drop('is_high_risk_road_condition', axis=1) # Drop the temporary flag

    # 2.5 speed_zone
    df['speed_zone'] = pd.cut(df['speed_limit'],
                              bins=[0, 45, np.inf],
                              labels=['low_speed', 'high_speed'],
                              right=True,
                              include_lowest=True)

    # 2.6 is_unsigned_urban_road
    df['is_unsigned_urban_road'] = ((df['road_type'] == 'urban') & (df['road_signs_present'] == False))

    # 2.7 is_accident_hotspot
    df['is_accident_hotspot'] = (df['num_reported_accidents'] >= 3)

    # 2.8 lane_category
    def get_lane_category(num_lanes):
        if num_lanes == 1:
            return 'single_lane'
        elif num_lanes == 2:
            return 'two_lanes'
        elif num_lanes == 3:
            return 'three_lanes'
        elif num_lanes >= 4:
            return 'multi_lanes'
        else:
            return np.nan # Should not happen

    df['lane_category'] = df['num_lanes'].apply(get_lane_category)

    # 2.9 is_urban_evening_clear_weather
    df['is_urban_evening_clear_weather'] = (
        (df['road_type'] == 'urban') &
        (df['time_of_day'] == 'evening') &
        (df['weather'] == 'clear')
    )

    # 2.10 accident_hotspot_x_weather
    df['accident_hotspot_x_weather'] = df.apply(
        lambda row: f"{ 'Hotspot' if row['is_accident_hotspot'] else 'NotHotspot'}_{row['weather']}",
        axis=1
    )

    # 2.11 low_speed_x_road_type_urban
    df['low_speed_x_road_type_urban'] = (
        (df['speed_zone'] == 'low_speed') &
        (df['road_type'] == 'urban')
    )

    # 2.12 low_speed_x_lane_category_two_lanes
    df['low_speed_x_lane_category_two_lanes'] = (
        (df['speed_zone'] == 'low_speed') &
        (df['lane_category'] == 'two_lanes')
    )

    # 2.13 low_speed_x_Visibility_Score
    df['low_speed_x_Visibility_Score'] = df.apply(
        lambda row: row['Visibility_Score'] if row['speed_zone'] == 'low_speed' else 0,
        axis=1
    )

    # 2.14 is_moderate_visibility_overprediction_zone
    df['is_moderate_visibility_overprediction_zone'] = df['Visibility_Score'].isin([1, 2])

    # Drop original columns that have been replaced or fully captured by new features
    columns_to_drop = [
        'id',
        'lighting', # Replaced by Visibility_Score
        'speed_limit', # Replaced by speed_zone
        'num_reported_accidents', # Replaced by is_accident_hotspot
        'num_lanes', # Replaced by lane_category
        'road_signs_present', # Used in is_unsigned_urban_road, but not kept itself
        'school_season', # Used in school_season_x_time_of_day, but not kept itself
        'holiday', # Used in holiday_x_time_of_day_x_bright, but not kept itself
        'curvature', # Used in high_risk_curvature_interaction, but not kept itself
        # 'weather' # Keep weather as it's used in accident_hotspot_x_weather and needs OHE
    ]

    df = df.drop(columns=columns_to_drop, errors='ignore')

    # 3. Final Encoding (to be done before modeling)
    # For now, we will just return the dataframe with the new features.
    # The final modeling script will handle one-hot encoding of categorical features.

    return df

if __name__ == "__main__":
    # Define paths
    raw_data_path = os.path.join('data', 'raw', 'train.csv')
    processed_data_path = os.path.join('data', 'processed', 'processed_train.csv')

    # Load raw data
    print(f"Loading raw data from {raw_data_path}...")
    raw_df = pd.read_csv(raw_data_path)

    # Build features
    print("Building features...")
    processed_df = build_features(raw_df.copy()) # Use a copy to avoid modifying raw_df

    # Save processed data
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    processed_df.to_csv(processed_data_path, index=False)
    print(f"Processed data saved to {processed_data_path}")
    print(f"Processed DataFrame shape: {processed_df.shape}")
    print("Processed DataFrame head:")
    print(processed_df.head())
