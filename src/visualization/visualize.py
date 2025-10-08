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
    df = pd.read_csv('train.csv')
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
                sns.boxplot(y=df[col], x=df['accident_risk'], order = df[col].value_counts().index)
        else: # numeric
            if col in ['num_lanes', 'num_reported_accidents', 'speed_limit']: # discrete numeric
                sns.boxplot(x=df[col], y=df['accident_risk'], order=sorted(df[col].unique()))
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


    # --- Generate HTML Report ---
    generate_html_report(plots_by_category)
    print(f"Visualizations and updated HTML report saved.")

except FileNotFoundError:
    print("Error: train.csv not found. Make sure the file is in the root directory.")
except Exception as e:
    print(f"An error occurred: {e}")
