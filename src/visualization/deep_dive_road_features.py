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
        <title>Deep Dive: Road Features and Accident Risk</title>
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
            <h1>Deep Dive: Road Features and Accident Risk</h1>
            <p>This report explores the complex relationships between road characteristics like speed limit, road type, and number of lanes, and their combined effect on accident risk.</p>
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
    with open('reports/deep_dive_road_features.html', 'w') as f:
        f.write(html_content)

# --- Main script ---
output_dir = 'reports/figures'
os.makedirs(output_dir, exist_ok=True)

plots = []

try:
    df = pd.read_csv('data/raw/train.csv')

    # --- 1. Speed Limit Distribution by Road Type ---
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=df, x='road_type', y='speed_limit', order=df.groupby('road_type')['speed_limit'].median().sort_values().index)
    plt.title('Speed Limit Distribution by Road Type')
    plt.xlabel('Road Type')
    plt.ylabel('Speed Limit (mph)')
    path = os.path.join('figures', 'speed_limit_by_road_type.png')
    plt.savefig(os.path.join(output_dir, 'speed_limit_by_road_type.png'))
    plt.close()
    plots.append({
        'title': 'Speed Limit Distribution by Road Type',
        'path': path,
        'explanation': 'This plot shows that highways have distinctly higher speed limits, while urban and rural roads have similar, lower speed limits. This confirms our initial hypothesis that speed limit is a strong indicator of road type.'
    })

    # --- 2. Accident Risk vs. Speed Limit, Faceted by Road Type ---
    plt.figure(figsize=(14, 7))
    g = sns.FacetGrid(df, col="road_type", height=5, aspect=1)
    g.map(sns.boxplot, "speed_limit", "accident_risk", order=sorted(df['speed_limit'].unique()))
    g.set_axis_labels("Speed Limit (mph)", "Accident Risk")
    g.set_titles("Road Type: {col_name}")
    plt.suptitle('Accident Risk vs. Speed Limit by Road Type', y=1.02)
    path = os.path.join('figures', 'risk_vs_speed_by_road_type.png')
    plt.savefig(os.path.join(output_dir, 'risk_vs_speed_by_road_type.png'))
    plt.close()
    plots.append({
        'title': 'Accident Risk vs. Speed Limit by Road Type',
        'path': path,
        'explanation': 'This plot reveals a critical interaction. The trend of higher risk at higher speeds is almost entirely driven by highways. For urban and rural roads, the relationship is much flatter. This is a strong signal for a potential interaction feature.'
    })

    # --- 3. Accident Risk vs. Number of Lanes, Faceted by Road Type (Line Plot) ---
    lanes_means = df.groupby(['road_type', 'num_lanes'])['accident_risk'].mean().reset_index()
    g = sns.FacetGrid(lanes_means, col="road_type", height=5, aspect=1, sharey=False)
    g.map(sns.lineplot, "num_lanes", "accident_risk", marker='o')
    g.set_axis_labels("Number of Lanes", "Mean Accident Risk")
    g.set_titles("Road Type: {col_name}")
    plt.suptitle('Mean Accident Risk vs. Number of Lanes by Road Type', y=1.02)
    path = os.path.join('figures', 'risk_vs_lanes_by_road_type.png')
    plt.savefig(os.path.join(output_dir, 'risk_vs_lanes_by_road_type.png'))
    plt.close()
    plots.append({
        'title': 'Accident Risk vs. Number of Lanes by Road Type',
        'path': path,
        'explanation': 'The number of lanes shows different effects depending on the road type. On highways, more lanes are associated with higher risk. On urban roads, the trend is less clear, but there might be a slight increase in risk with more lanes. This suggests another valuable interaction.'
    })
    
    # --- 4. Curvature vs Accident Risk, Faceted by Road Type ---
    df_sample = df.sample(n=5000, random_state=42)
    g = sns.FacetGrid(df_sample, col="road_type", height=5, aspect=1)
    g.map(sns.scatterplot, "curvature", "accident_risk", alpha=0.5)
    g.set_axis_labels("Curvature", "Accident Risk")
    g.set_titles("Road Type: {col_name}")
    plt.suptitle('Accident Risk vs. Curvature by Road Type', y=1.02)
    path = os.path.join('figures', 'risk_vs_curvature_by_road_type.png')
    plt.savefig(os.path.join(output_dir, 'risk_vs_curvature_by_road_type.png'))
    plt.close()
    plots.append({
        'title': 'Accident Risk vs. Curvature by Road Type',
        'path': path,
        'explanation': 'This plot shows the relationship between road curvature and accident risk, broken down by road type. Across all types, higher curvature tends to be associated with slightly higher accident risk, but the effect is most pronounced on highways and rural roads.'
    })

    # --- 5. Accident Risk vs. Public Road ---
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='public_road', y='accident_risk')
    plt.title('Accident Risk: Public vs. Private Roads')
    plt.xlabel('Is Public Road')
    plt.ylabel('Accident Risk')
    path = os.path.join('figures', 'public_road_analysis.png')
    plt.savefig(os.path.join(output_dir, 'public_road_analysis.png'))
    plt.close()
    plots.append({
        'title': 'Accident Risk: Public vs. Private Roads',
        'path': path,
        'explanation': 'This plot tests the hypothesis that public roads are more dangerous than private roads. It compares the distribution of accident risk for each category.'
    })

    # --- 5. Violin Plot for Accident Risk vs. Road Type ---
    plt.figure(figsize=(12, 7))
    sns.violinplot(data=df, x='road_type', y='accident_risk', order=df.groupby('road_type')['accident_risk'].median().sort_values().index)
    plt.title('Accident Risk Distribution by Road Type (Violin Plot)')
    plt.xlabel('Road Type')
    plt.ylabel('Accident Risk')
    path = os.path.join('figures', 'ar_vs_road_type_violin.png')
    plt.savefig(os.path.join(output_dir, 'ar_vs_road_type_violin.png'))
    plt.close()
    plots.append({
        'title': 'Accident Risk vs. Road Type (Violin Plot)',
        'path': path,
        'explanation': 'This violin plot shows the probability density of accident risk for each road type. We can see the shape of the distribution, not just the summary statistics. This confirms that highways have a higher median accident risk and a wider distribution of risk values.'
    })

    # --- 6. Hexbin Plot for Accident Risk vs. Curvature ---
    df_sample_adv = df.sample(n=10000, random_state=42)
    plt.figure(figsize=(10, 8))
    plt.hexbin(df_sample_adv['curvature'], df_sample_adv['accident_risk'], gridsize=50, cmap='viridis')
    plt.colorbar(label='Count in Bin')
    plt.title('Accident Risk vs. Curvature (Hexbin Plot)')
    plt.xlabel('Curvature')
    plt.ylabel('Accident Risk')
    path = os.path.join('figures', 'ar_vs_curvature_hexbin.png')
    plt.savefig(os.path.join(output_dir, 'ar_vs_curvature_hexbin.png'))
    plt.close()
    plots.append({
        'title': 'Accident Risk vs. Curvature (Hexbin Plot)',
        'path': path,
        'explanation': 'This hexbin plot shows the density of data points for curvature and accident risk. The darkest areas indicate the most common combinations. We can see that the highest concentration of points is at low curvature and low-to-medium accident risk.'
    })

    # --- 7. Faceted Regression Plots for Curvature vs. Accident Risk ---
    g = sns.lmplot(data=df_sample_adv, x='curvature', y='accident_risk', col='road_type', hue='road_type', height=5, aspect=1, scatter_kws={'alpha':0.3})
    g.set_axis_labels("Curvature", "Accident Risk")
    g.set_titles("Road Type: {col_name}")
    plt.suptitle('Accident Risk vs. Curvature with Regression Line by Road Type', y=1.02)
    path = os.path.join('figures', 'ar_vs_curvature_regression_faceted.png')
    plt.savefig(os.path.join(output_dir, 'ar_vs_curvature_regression_faceted.png'))
    plt.close()
    plots.append({
        'title': 'Accident Risk vs. Curvature with Regression Line by Road Type',
        'path': path,
        'explanation': 'This plot shows the relationship between curvature and accident risk for each road type, with a fitted regression line. This makes the trend clearer. We can see a positive correlation for all road types, but the slope of the line (the strength of the relationship) is slightly different for each. This reinforces the idea that interaction features could be valuable.'
    })


    # --- Generate HTML Report ---
    generate_html_report(plots)
    print(f"Deep dive report and visualizations for road features saved.")

except FileNotFoundError:
    print("Error: data/raw/train.csv not found. Make sure the file is in the correct directory.")
except Exception as e:
    print(f"An error occurred: {e}")
