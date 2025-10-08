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
    df = pd.read_csv('train.csv')

    # --- Holiday vs. Time of Day ---
    plt.figure(figsize=(12, 7))
    sns.pointplot(data=df, x='time_of_day', y='accident_risk', hue='holiday', order=['morning', 'afternoon', 'evening', 'night'])
    plt.title('Accident Risk by Time of Day, Comparing Holidays vs. Non-Holidays')
    plt.xlabel('Time of Day')
    plt.ylabel('Mean Accident Risk')
    plot_path_on_disk = os.path.join(output_dir, 'holiday_vs_time_of_day.png')
    plt.savefig(plot_path_on_disk)
    plt.close()
    plots_info.append({
        'title': 'Holiday vs. Time of Day',
        'path': os.path.join('figures', 'holiday_vs_time_of_day.png'),
        'explanation': 'This plot compares the mean accident risk throughout the day for holidays vs. non-holidays. We can look for differences in the daily pattern. For example, is the morning peak in accident risk (if any) different on holidays when there is no morning commute?'
    })

    # --- School Season vs. Time of Day ---
    plt.figure(figsize=(12, 7))
    sns.pointplot(data=df, x='time_of_day', y='accident_risk', hue='school_season', order=['morning', 'afternoon', 'evening', 'night'])
    plt.title('Accident Risk by Time of Day, Comparing School Season vs. Non-School Season')
    plt.xlabel('Time of Day')
    plt.ylabel('Mean Accident Risk')
    plot_path_on_disk = os.path.join(output_dir, 'school_season_vs_time_of_day.png')
    plt.savefig(plot_path_on_disk)
    plt.close()
    plots_info.append({
        'title': 'School Season vs. Time of Day',
        'path': os.path.join('figures', 'school_season_vs_time_of_day.png'),
        'explanation': 'This plot compares the mean accident risk throughout the day for school season vs. non-school season. We can look for differences in the daily pattern, especially during morning and afternoon times, which correspond to school-related traffic.'
    })

    # --- Generate HTML Report ---
    generate_deep_dive_html(plots_info)
    print(f"Temporal deep dive visualization and HTML report created successfully.")

except FileNotFoundError:
    print("Error: train.csv not found. Make sure the file is in the root directory.")
except Exception as e:
    print(f"An error occurred: {e}")
