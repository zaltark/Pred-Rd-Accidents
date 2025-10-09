import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def generate_html_report(plots_by_category):
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Exploratory Data Analysis Report</title>
        <style>
            body { font-family: sans-serif; margin: 2em; }
            h1, h2, h3 { color: #333; }
            img { border: 1px solid #ddd; border-radius: 4px; padding: 5px; max-width: 100%; height: auto; }
            .container { max-width: 800px; margin: auto; }
            .plot { margin-bottom: 2em; }
            .category { margin-bottom: 3em; border-bottom: 2px solid #ccc; padding-bottom: 2em; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Exploratory Data Analysis Report</h1>
    """

    for category, plots in plots_by_category.items():
        html_content += f"""
        <div class="category">
            <h2>{category}</h2>
        """
        for plot in plots:
            html_content += f"""
            <div class="plot">
                <h3>{plot['title']}</h3>
                <img src="{plot['path']}" alt="{plot['title']}">
                <p>{plot['explanation']}</p>
            </div>
            """
        if category == "Environmental Conditions":
            html_content += """
            <div class="plot">
                <h3>Deep Dive Analysis</h3>
                <p>For a more detailed look at the interaction of environmental features, see the <a href="deep_dive_environmental.html">Deep Dive: Environmental Factors and Accident Risk</a> report.</p>
            </div>
            """
        if category == "Road Characteristics":
            html_content += """
            <div class="plot">
                <h3>Deep Dive Analysis</h3>
                <p>For a more detailed look at the interaction of road features, see the <a href="deep_dive_road_features.html">Deep Dive: Road Features and Accident Risk</a> report.</p>
            </div>
            """
        if category == "Safety & Historical Factors":
            html_content += """
            <div class="plot">
                <h3>Deep Dive Analysis</h3>
                <p>For a more detailed look at the effect of road signs, see the <a href="deep_dive_road_signs.html">Deep Dive: Road Signs and Accident Risk</a> report.</p>
            </div>
            """
        if category == "Temporal Factors":
            html_content += """
            <div class="plot">
                <h3>Deep Dive Analysis</h3>
                <p>For a more detailed look at the interaction of temporal features, see the <a href="deep_dive_temporal.html">Deep Dive: Temporal Features and Accident Risk</a> report.</p>
            </div>
            """
        html_content += "</div>"


    html_content += """
        </div>
    </body>
    </html>
    """
    with open('reports/eda_report.html', 'w') as f:
        f.write(html_content)

def plot_lighting_time_heatmap(df, output_dir):
    """Generates and saves a heatmap of accident risk by lighting and time of day."""
    plt.figure(figsize=(12, 8))
    pivot_table = df.pivot_table(values='accident_risk', index='lighting', columns='time_of_day', aggfunc='mean')
    sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="YlGnBu")
    plt.title('Mean Accident Risk by Lighting and Time of Day')
    path = os.path.join('figures', 'lighting_vs_time_of_day.png')
    plt.savefig(os.path.join(output_dir, 'lighting_vs_time_of_day.png'))
    plt.close()
    return {
        'title': 'Mean Accident Risk by Lighting and Time of Day',
        'path': path,
        'explanation': 'This heatmap shows the interaction between lighting conditions and time of day on accident risk.'
    }

def plot_bright_evening_comparison(df, output_dir):
    """Generates a bar chart comparing accident risk for bright lighting across different times of the day."""
    plt.figure(figsize=(10, 6))
    bright_df = df[df['lighting'] == 'Bright']
    sns.barplot(data=bright_df, x='time_of_day', y='accident_risk', order=['morning', 'afternoon', 'evening', 'night'])
    plt.title('Mean Accident Risk for "Bright" Lighting by Time of Day')
    plt.ylabel('Mean Accident Risk')
    plt.xlabel('Time of Day')
    path = os.path.join('figures', 'bright_evening_comparison.png')
    plt.savefig(os.path.join(output_dir, 'bright_evening_comparison.png'))
    plt.close()
    return {
        'title': 'Mean Accident Risk for "Bright" Lighting',
        'path': path,
        'explanation': 'This chart compares the mean accident risk during "Bright" lighting conditions across different times of the day, highlighting the risk in the evening.'
    }

def plot_bright_time_school_season_comparison(df, output_dir):
    """Generates a grouped bar chart for bright afternoon/evening risk by school season."""
    plt.figure(figsize=(10, 6))
    filtered_df = df[(df['lighting'] == 'Bright') & (df['time_of_day'].isin(['afternoon', 'evening']))]
    sns.barplot(data=filtered_df, x='time_of_day', y='accident_risk', hue='school_season', order=['afternoon', 'evening'])
    plt.title('Mean Accident Risk for Bright Afternoon/Evening by School Season')
    plt.ylabel('Mean Accident Risk')
    plt.xlabel('Time of Day')
    path = os.path.join('figures', 'bright_time_school_season_comparison.png')
    plt.savefig(os.path.join(output_dir, 'bright_time_school_season_comparison.png'))
    plt.close()
    return {
        'title': 'Bright Afternoon/Evening Risk by School Season',
        'path': path,
        'explanation': 'This chart compares accident risk in bright afternoon and evening, grouped by whether it is school season or not.'
    }

def plot_bright_time_holiday_comparison(df, output_dir):
    """Generates a grouped bar chart for bright afternoon/evening risk by holiday."""
    plt.figure(figsize=(10, 6))
    filtered_df = df[(df['lighting'] == 'Bright') & (df['time_of_day'].isin(['afternoon', 'evening']))]
    sns.barplot(data=filtered_df, x='time_of_day', y='accident_risk', hue='holiday', order=['afternoon', 'evening'])
    plt.title('Mean Accident Risk for Bright Afternoon/Evening by Holiday')
    plt.ylabel('Mean Accident Risk')
    plt.xlabel('Time of Day')
    path = os.path.join('figures', 'bright_time_holiday_comparison.png')
    plt.savefig(os.path.join(output_dir, 'bright_time_holiday_comparison.png'))
    plt.close()
    return {
        'title': 'Bright Afternoon/Evening Risk by Holiday',
        'path': path,
        'explanation': 'This chart compares accident risk in bright afternoon and evening, grouped by whether it is a holiday or not.'
    }


# --- Main script ---
output_dir = 'reports/figures'
os.makedirs(output_dir, exist_ok=True)

# Define feature categories
road_characteristics = ['road_type', 'num_lanes', 'curvature', 'speed_limit', 'public_road']
environmental_conditions = ['lighting', 'weather']
temporal_factors = ['time_of_day', 'holiday', 'school_season']
safety_historical_factors = ['road_signs_present', 'num_reported_accidents']

all_features = road_characteristics + environmental_conditions + temporal_factors + safety_historical_factors
plots_by_category = {
    "Road Characteristics": [],
    "Environmental Conditions": [],
    "Temporal Factors": [],
    "Safety & Historical Factors": [],
    "Other": []
}

def get_category(col):
    if col in road_characteristics:
        return "Road Characteristics"
    elif col in environmental_conditions:
        return "Environmental Conditions"
    elif col in temporal_factors:
        return "Temporal Factors"
    elif col in safety_historical_factors:
        return "Safety & Historical Factors"
    else:
        return "Other"

try:
    df = pd.read_csv('data/raw/train.csv')
    # Map lighting categories to new luminosity names
    lighting_map = {
        'daylight': 'Bright',
        'dim': 'Dim',
        'night': 'Dark'
    }
    df['lighting'] = df['lighting'].replace(lighting_map)
    df_sample = df.sample(n=1000, random_state=42)

    # --- Visualize All Features ---
    for col in all_features:
        category = get_category(col)
        
        # Distribution plot
        plt.figure(figsize=(10, 6))
        if df[col].dtype == 'object' or df[col].dtype == 'bool':
            sns.countplot(y=df[col], order = df[col].value_counts().index)
        else: # numeric
            if col in ['num_lanes', 'num_reported_accidents', 'speed_limit']: # discrete numeric
                sns.countplot(x=df[col])
            else: # continuous numeric
                sns.histplot(df[col], kde=True)

        plt.title(f'Distribution of {col}')
        plt.tight_layout()
        path = os.path.join('figures', f'{col}_distribution.png')
        plt.savefig(os.path.join(output_dir, f'{col}_distribution.png'))
        plt.close()
        plots_by_category[category].append({
            'title': f'Distribution of {col}',
            'path': path,
            'explanation': f'This plot shows the distribution of the {col} feature.'
        })

        # Relationship with accident_risk
        plt.figure(figsize=(10, 6))
        if df[col].dtype == 'object' or df[col].dtype == 'bool':
            if df[col].dtype == 'bool':
                sns.boxplot(y=df[col].astype(str), x=df['accident_risk'])
            else:
                sns.boxplot(y=df[col], x=df['accident_risk'])
        else: # numeric
            if col in ['num_lanes', 'num_reported_accidents', 'speed_limit']: # discrete numeric
                sns.boxplot(x=df[col], y=df['accident_risk'])
            else: # continuous numeric
                sns.scatterplot(x=df_sample[col], y=df_sample['accident_risk'])

        plt.title(f'{col} vs. Accident Risk')
        plt.tight_layout()
        path = os.path.join('figures', f'{col}_vs_accident_risk.png')
        plt.savefig(os.path.join(output_dir, f'{col}_vs_accident_risk.png'))
        plt.close()
        plots_by_category[category].append({
            'title': f'{col} vs. Accident Risk',
            'path': path,
            'explanation': f'This plot shows the relationship between {col} and accident risk.'
        })

    # --- Custom Visualizations ---
    lighting_time_plot = plot_lighting_time_heatmap(df, output_dir)
    plots_by_category["Environmental Conditions"].append(lighting_time_plot)

    bright_evening_plot = plot_bright_evening_comparison(df, output_dir)
    plots_by_category["Environmental Conditions"].append(bright_evening_plot)

    bright_time_school_season_plot = plot_bright_time_school_season_comparison(df, output_dir)
    plots_by_category["Temporal Factors"].append(bright_time_school_season_plot)

    bright_time_holiday_plot = plot_bright_time_holiday_comparison(df, output_dir)
    plots_by_category["Temporal Factors"].append(bright_time_holiday_plot)


    # --- Generate HTML Report ---
    generate_html_report(plots_by_category)
    print(f"Visualizations and updated HTML report saved.")

except FileNotFoundError:
    print("Error: train.csv not found. Make sure the file is in the root directory.")
except Exception as e:
    print(f"An error occurred: {e}")