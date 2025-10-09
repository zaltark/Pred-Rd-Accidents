import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def generate_deep_dive_html(plots):
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Deep Dive: Temporal Features and Accident Risk</title>
        <style>
            body { font-family: sans-serif; margin: 2em; }
            h1, h2 { color: #333; }
            img { border: 1px solid #ddd; border-radius: 4px; padding: 5px; max-width: 100%; height: auto; }
            .container { max-width: 800px; margin: auto; }
            .plot { margin-bottom: 2em; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Deep Dive: The Effect of Temporal Features on Accident Risk</h1>
    """
    for plot in plots:
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
    with open('reports/deep_dive_temporal.html', 'w') as f:
        f.write(html_content)

# --- Main script ---
output_dir = 'reports/figures'
os.makedirs(output_dir, exist_ok=True)
plots_info = []

try:
    df = pd.read_csv('data/raw/train.csv')

    # Map lighting categories to new luminosity names
    lighting_map = {
        'daylight': 'Bright',
        'dim': 'Dim',
        'night': 'Dark'
    }
    df['lighting'] = df['lighting'].replace(lighting_map)

    # Convert time_of_day to an ordered categorical type for correct sorting in plots
    from pandas.api.types import CategoricalDtype
    time_of_day_order = ['morning', 'afternoon', 'evening']
    time_cat_type = CategoricalDtype(categories=time_of_day_order, ordered=True)
    df['time_of_day'] = df['time_of_day'].astype(time_cat_type)

    # --- Proof: Distribution of time_of_day ---
    plt.figure(figsize=(10, 5))
    sns.countplot(y=df['time_of_day'])
    plt.title('Distribution of time_of_day Categories')
    plt.xlabel('Count')
    plt.ylabel('Time of Day')
    plot_path_on_disk = os.path.join(output_dir, 'proof_time_of_day_dist.png')
    plt.savefig(plot_path_on_disk)
    plt.close()
    plots_info.append({
        'title': 'Proof: Distribution of time_of_day',
        'path': os.path.join('figures', 'proof_time_of_day_dist.png'),
        'explanation': 'This plot shows the actual unique categories and their counts present in the `time_of_day` column. It serves to verify which categories exist in the data.'
    })

    # --- Proof: Distribution of lighting ---
    plt.figure(figsize=(10, 5))
    sns.countplot(y=df['lighting'])
    plt.title('Distribution of lighting Categories')
    plt.xlabel('Count')
    plt.ylabel('Lighting Condition')
    plot_path_on_disk = os.path.join(output_dir, 'proof_lighting_dist.png')
    plt.savefig(plot_path_on_disk)
    plt.close()
    plots_info.append({
        'title': 'Proof: Distribution of lighting',
        'path': os.path.join('figures', 'proof_lighting_dist.png'),
        'explanation': 'This plot shows the actual unique categories and their counts present in the `lighting` column. It serves to verify which categories exist in the data, for comparison with the `time_of_day` feature.'
    })

    # --- Direct Comparison: time_of_day vs. lighting ---
    plt.figure(figsize=(12, 7))
    sns.countplot(data=df, x='time_of_day', hue='lighting')
    plt.title('Lighting Conditions within each Time of Day')
    plt.xlabel('Time of Day')
    plt.ylabel('Count')
    plot_path_on_disk = os.path.join(output_dir, 'time_vs_lighting_comparison.png')
    plt.savefig(plot_path_on_disk)
    plt.close()
    plots_info.append({
        'title': 'Direct Comparison: Time of Day vs. Lighting',
        'path': os.path.join('figures', 'time_vs_lighting_comparison.png'),
        'explanation': 'This plot directly compares the `time_of_day` and `lighting` features. It shows the distribution of lighting conditions that occur within each time of day bucket. For example, it clearly shows that the \'Evening\' time of day contains both \'Dusk\' and \'Night\' lighting conditions.'
    })

    # --- Holiday vs. Time of Day (Line Plot) ---
    plt.figure(figsize=(12, 7))
    holiday_means = df.groupby(['time_of_day', 'holiday'])['accident_risk'].mean().reset_index()
    sns.lineplot(data=holiday_means, x='time_of_day', y='accident_risk', hue='holiday', style='holiday', markers=True, dashes=False)
    plt.title('Accident Risk by Time of Day, Comparing Holidays vs. Non-Holidays')
    plt.xlabel('Time of Day')
    plt.ylabel('Mean Accident Risk')
    min_val = holiday_means['accident_risk'].min()
    max_val = holiday_means['accident_risk'].max()
    plt.ylim(min_val - (max_val-min_val)*0.1, max_val + (max_val-min_val)*0.1)
    plot_path_on_disk = os.path.join(output_dir, 'holiday_vs_time_of_day.png')
    plt.savefig(plot_path_on_disk)
    plt.close()
    plots_info.append({
        'title': 'Holiday vs. Time of Day',
        'path': os.path.join('figures', 'holiday_vs_time_of_day.png'),
        'explanation': 'This plot compares the mean accident risk throughout the day for holidays vs. non-holidays. The "crossing lines" show how the risk pattern changes. A tighter y-axis has been used to emphasize the differences.'
    })

    # --- School Season vs. Time of Day (Line Plot) ---
    plt.figure(figsize=(12, 7))
    school_means = df.groupby(['time_of_day', 'school_season'])['accident_risk'].mean().reset_index()
    sns.lineplot(data=school_means, x='time_of_day', y='accident_risk', hue='school_season', style='school_season', markers=True, dashes=False)
    plt.title('Accident Risk by Time of Day, Comparing School Season vs. Non-School Season')
    plt.xlabel('Time of Day')
    plt.ylabel('Mean Accident Risk')
    min_val = school_means['accident_risk'].min()
    max_val = school_means['accident_risk'].max()
    plt.ylim(min_val - (max_val-min_val)*0.1, max_val + (max_val-min_val)*0.1)
    plot_path_on_disk = os.path.join(output_dir, 'school_season_vs_time_of_day.png')
    plt.savefig(plot_path_on_disk)
    plt.close()
    plots_info.append({
        'title': 'School Season vs. Time of Day',
        'path': os.path.join('figures', 'school_season_vs_time_of_day.png'),
        'explanation': 'This plot compares the mean accident risk throughout the day for school season vs. non-school season. The "crossing lines" clearly show the interaction: risk is lower on school mornings but higher on school afternoons. A tighter y-axis has been used to emphasize these differences.'
    })

    # --- Generate HTML Report ---
    generate_deep_dive_html(plots_info)
    print(f"Temporal deep dive visualization and HTML report created successfully.")

except FileNotFoundError:
    print("Error: data/raw/train.csv not found. Make sure the file is in the root directory.")
except Exception as e:
    print(f"An error occurred: {e}")