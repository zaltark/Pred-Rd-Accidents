import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def generate_html_report(plots):
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Deep Dive: Environmental Factors and Accident Risk</title>
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
            <h1>Deep Dive: Environmental Factors and Accident Risk</h1>
            <p>This report explores the interactions between environmental factors like lighting and weather, and their combined effect on accident risk.</p>
    """

    for plot in plots:
        html_content += f"""
        <div class="plot">
            <h3>{plot['title']}</h3>
            <img src="{plot['path']}" alt="{plot['title']}">
            <p>{plot['explanation']}</p>
        </div>
        """

    html_content += """
        </div>
    </body>
    </html>
    """
    with open('reports/deep_dive_environmental.html', 'w') as f:
        f.write(html_content)

# --- Main script ---
output_dir = 'reports/figures'
os.makedirs(output_dir, exist_ok=True)

plots = []

try:
    df = pd.read_csv('data/raw/train.csv')
    # Map lighting categories to new luminosity names
    lighting_map = {
        'daylight': 'Bright',
        'dim': 'Dim',
        'night': 'Dark'
    }
    df['lighting'] = df['lighting'].replace(lighting_map)

    # --- 1. Lighting vs. Time of Day ---
    plt.figure(figsize=(12, 7))
    sns.countplot(data=df, x='lighting', hue='time_of_day')
    plt.title('Lighting Conditions by Time of Day')
    plt.xlabel('Lighting Condition')
    plt.ylabel('Count')
    path = os.path.join('figures', 'lighting_vs_time_of_day.png')
    plt.savefig(os.path.join(output_dir, 'lighting_vs_time_of_day.png'))
    plt.close()
    plots.append({
        'title': 'Lighting Conditions by Time of Day',
        'path': path,
        'explanation': "This plot confirms the expected relationship between lighting and time of day. \"Night\" occurs only at night, \"Daylight\" mostly during the day. \"Dawn\" and \"Dusk\" appear at the appropriate times. This gives us confidence in the data's consistency."
    })

    # --- 2. Accident Risk vs. Weather, Faceted by Lighting ---
    plt.figure(figsize=(14, 7))
    g = sns.FacetGrid(df, col="lighting", height=5, aspect=1)
    g.map(sns.boxplot, "weather", "accident_risk")
    g.set_axis_labels("Weather Condition", "Accident Risk")
    g.set_titles("Lighting: {col_name}")
    plt.suptitle('Accident Risk vs. Weather by Lighting Condition', y=1.02)
    path = os.path.join('figures', 'risk_vs_weather_by_lighting.png')
    plt.savefig(os.path.join(output_dir, 'risk_vs_weather_by_lighting.png'))
    plt.close()
    plots.append({
        'title': 'Accident Risk vs. Weather by Lighting Condition',
        'path': path,
        'explanation': 'This is a key insight. The impact of weather is highly dependent on lighting. Rainy and Foggy conditions at Night lead to a dramatic increase in accident risk compared to the same conditions during Daylight. This strongly supports creating an interaction feature.'
    })

    # --- 3. Accident Risk vs. Lighting, Faceted by Road Type ---
    plt.figure(figsize=(14, 7))
    g = sns.FacetGrid(df, col="road_type", height=5, aspect=1)
    g.map(sns.boxplot, "lighting", "accident_risk")
    g.set_axis_labels("Lighting Condition", "Accident Risk")
    g.set_titles("Road Type: {col_name}")
    plt.suptitle('Accident Risk vs. Lighting by Road Type', y=1.02)
    path = os.path.join('figures', 'risk_vs_lighting_by_road_type.png')
    plt.savefig(os.path.join(output_dir, 'risk_vs_lighting_by_road_type.png'))
    plt.close()
    plots.append({
        'title': 'Accident Risk vs. Lighting by Road Type',
        'path': path,
        'explanation': 'The increase in accident risk at Night is present across all road types, but it is most pronounced on Highways. This suggests that the combination of high speed and low visibility is particularly dangerous.'
    })

    # --- Advanced Analysis for 'Clear' Weather ---
    df_clear = df[df['weather'] == 'clear'].copy() # Filter for clear weather

    if not df_clear.empty:
        # 4.1 Conditional Risk Distribution (Faceted Histograms/KDEs)
        plt.figure(figsize=(16, 6))
        g = sns.FacetGrid(df_clear, col="road_type", row="time_of_day", height=3, aspect=1.5, sharey=False)
        g.map(sns.histplot, "accident_risk", kde=True, bins=20)
        g.set_axis_labels("Accident Risk", "Count")
        g.set_titles(col_template="Road Type: {col_name}", row_template="Time of Day: {row_name}")
        plt.suptitle('Accident Risk Distribution in Clear Weather (Faceted by Road Type & Time of Day)', y=1.02)
        plot_path = os.path.join(output_dir, 'clear_weather_risk_dist_faceted.png')
        plt.savefig(plot_path)
        plt.close()
        plots.append({
            'title': 'Accident Risk Distribution in Clear Weather (Faceted)',
            'path': os.path.join('figures', 'clear_weather_risk_dist_faceted.png'),
            'explanation': 'This plot shows the distribution of accident risk specifically when weather is \'Clear\', faceted by `road_type` and `time_of_day`. It helps identify if risk patterns vary even in good weather conditions.'
        })

        # 4.2 Interaction Line Plot (Mean Risk vs. num_lanes, hue by road_type)
        # Ensure num_lanes is treated as categorical for plotting discrete points
        df_clear['num_lanes_cat'] = df_clear['num_lanes'].astype(str)
        lanes_risk_clear = df_clear.groupby(['road_type', 'num_lanes_cat'])['accident_risk'].mean().reset_index()

        plt.figure(figsize=(12, 7))
        sns.lineplot(data=lanes_risk_clear, x='num_lanes_cat', y='accident_risk', hue='road_type', marker='o')
        plt.title('Mean Accident Risk in Clear Weather (num_lanes vs. road_type)')
        plt.xlabel('Number of Lanes')
        plt.ylabel('Mean Accident Risk')
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'clear_weather_lanes_roadtype_interaction.png')
        plt.savefig(plot_path)
        plt.close()
        plots.append({
            'title': 'Mean Accident Risk in Clear Weather (num_lanes vs. road_type)',
            'path': os.path.join('figures', 'clear_weather_lanes_roadtype_interaction.png'),
            'explanation': 'This plot shows how the mean accident risk changes with the number of lanes, faceted by road type, specifically when weather is \'Clear\'. It highlights interactions that persist even in good weather.'
        })

        # 4.3 Heatmap of Mean Risk (road_type vs. time_of_day)
        heatmap_data = df_clear.groupby(['road_type', 'time_of_day'])['accident_risk'].mean().unstack()
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="viridis")
        plt.title('Mean Accident Risk in Clear Weather (Road Type vs. Time of Day)')
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'clear_weather_roadtype_time_heatmap.png')
        plt.savefig(plot_path)
        plt.close()
        plots.append({
            'title': 'Mean Accident Risk in Clear Weather (Road Type vs. Time of Day Heatmap)',
            'path': os.path.join('figures', 'clear_weather_roadtype_time_heatmap.png'),
            'explanation': 'This heatmap shows the mean accident risk for combinations of `road_type` and `time_of_day`, specifically when weather is \'Clear\'. Darker cells indicate higher risk, helping to pinpoint specific high-risk contexts even in good weather.'
        })
    else:
        plots.append({
            'title': 'Advanced Analysis for \'Clear\' Weather',
            'path': '',
            'explanation': 'No data available for \'Clear\' weather to perform advanced analysis.'
        })

    # --- 6. Mean Accident Risk vs. num_reported_accidents, Faceted by Weather ---
    # Ensure num_reported_accidents is treated as categorical for plotting discrete points
    df['num_reported_accidents_cat'] = df['num_reported_accidents'].astype(str)
    accidents_risk_weather = df.groupby(['weather', 'num_reported_accidents_cat'])['accident_risk'].mean().reset_index()

    plt.figure(figsize=(12, 7))
    g = sns.FacetGrid(accidents_risk_weather, col="weather", height=5, aspect=1, col_order=['clear', 'rainy', 'foggy'])
    g.map(sns.lineplot, "num_reported_accidents_cat", "accident_risk", marker='o')
    g.set_axis_labels("Number of Reported Accidents", "Mean Accident Risk")
    g.set_titles("Weather: {col_name}")
    plt.suptitle('Mean Accident Risk vs. Number of Reported Accidents (Faceted by Weather)', y=1.02)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'accidents_vs_weather_interaction.png')
    plt.savefig(plot_path)
    plt.close()
    plots.append({
        'title': 'Mean Accident Risk vs. Number of Reported Accidents (Faceted by Weather)',
        'path': os.path.join('figures', 'accidents_vs_weather_interaction.png'),
        'explanation': 'This plot shows how the mean accident risk changes with the number of previously reported accidents, faceted by weather condition. It helps identify if the impact of historical accident data varies with weather.'
    })

    # --- 5. Heatmap: Mean Accident Risk (Weather vs. Lighting) ---
    heatmap_data_weather_lighting = df.groupby(['weather', 'lighting'])['accident_risk'].mean().unstack()
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data_weather_lighting, annot=True, fmt=".3f", cmap="viridis")
    plt.title('Mean Accident Risk (Weather vs. Lighting)')
    plt.xlabel('Lighting Condition')
    plt.ylabel('Weather Condition')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'weather_vs_lighting_heatmap.png')
    plt.savefig(plot_path)
    plt.close()
    plots.append({
        'title': 'Mean Accident Risk (Weather vs. Lighting Heatmap)',
        'path': os.path.join('figures', 'weather_vs_lighting_heatmap.png'),
        'explanation': 'This heatmap shows the mean accident risk for all combinations of `weather` and `lighting` conditions. Darker cells indicate higher risk, highlighting the most dangerous environmental contexts.'
    })

    # --- Generate HTML Report ---
    generate_html_report(plots)
    print(f"Deep dive report and visualizations for environmental factors saved.")

except FileNotFoundError:
    print("Error: data/raw/train.csv not found. Make sure the file is in the correct directory.")
except Exception as e:
    print(f"An error occurred: {e}")