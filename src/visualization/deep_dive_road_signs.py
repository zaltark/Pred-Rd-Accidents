import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def generate_deep_dive_html(box_plot_path, point_plot_path, explanation, median_table):
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Deep Dive: Road Signs and Accident Risk</title>
        <style>
            body {{ font-family: sans-serif; margin: 2em; }}
            h1, h2 {{ color: #333; }}
            img {{ border: 1px solid #ddd; border-radius: 4px; padding: 5px; max-width: 100%; height: auto; }}
            .container {{ max-width: 800px; margin: auto; }}
            .plot {{ margin-bottom: 2em; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Deep Dive: The Effect of Road Signs on Accident Risk</h1>
            
            <h2>Median Accident Risk</h2>
            {median_table}
            <p>The table above shows the precise median accident risk for each scenario. This allows us to quantify the differences that may not be obvious from the plots.</p>

            <div class="plot">
                <h2>Direct Comparison of Mean Accident Risk</h2>
                <img src="{point_plot_path}" alt="Faceted Point Plot of Road Signs vs. Accident Risk">
                <p>This plot shows the mean accident risk (the point) and the 95% confidence interval (the vertical line) for each scenario. This type of plot is excellent for directly comparing the central tendency between categories and determining if the differences are statistically significant.</p>
            </div>

            <div class="plot">
                <h2>Distribution of Accident Risk (Box Plot)</h2>
                <img src="{box_plot_path}" alt="Faceted Box Plot of Road Signs vs. Accident Risk">
                <p>{explanation}</p>
            </div>
        </div>
    </body>
    </html>
    """
    with open('reports/deep_dive_road_signs.html', 'w') as f:
        f.write(html_content)

# --- Main script ---
output_dir = 'reports/figures'
os.makedirs(output_dir, exist_ok=True)

try:
    df = pd.read_csv('train.csv')

    # --- Create the box plot ---
    g_box = sns.catplot(
        data=df, x='road_signs_present', y='accident_risk',
        col='road_type', kind='box', height=6, aspect=0.8
    )
    g_box.fig.suptitle('Distribution of Accident Risk, by Road Type and Sign Presence', y=1.03)
    g_box.set_axis_labels("Road Signs Present", "Accident Risk")
    box_plot_path_on_disk = os.path.join(output_dir, 'road_signs_faceted_by_road_type_box.png')
    plt.savefig(box_plot_path_on_disk)
    plt.close()

    # --- Create the point plot ---
    g_point = sns.catplot(
        data=df, x='road_signs_present', y='accident_risk',
        col='road_type', kind='point', height=6, aspect=0.8
    )
    g_point.fig.suptitle('Mean Accident Risk, by Road Type and Sign Presence', y=1.03)
    g_point.set_axis_labels("Road Signs Present", "Accident Risk")
    point_plot_path_on_disk = os.path.join(output_dir, 'road_signs_faceted_by_road_type_point.png')
    plt.savefig(point_plot_path_on_disk)
    plt.close()

    # --- Calculate median values ---
    median_risk = df.groupby(['road_type', 'road_signs_present'])['accident_risk'].median().unstack()
    median_table_html = median_risk.to_html()


    # --- Generate the HTML report ---
    box_plot_path_for_html = os.path.join('figures', 'road_signs_faceted_by_road_type_box.png')
    point_plot_path_for_html = os.path.join('figures', 'road_signs_faceted_by_road_type_point.png')
    explanation = """
    <p>This plot shows the distribution of accident risk for each scenario. While it gives a good sense of the overall spread, the differences in the median can be hard to discern.</p>
    """
    generate_deep_dive_html(box_plot_path_for_html, point_plot_path_for_html, explanation, median_table_html)

    print(f"Deep dive visualization and HTML report updated successfully.")

except FileNotFoundError:
    print("Error: train.csv not found. Make sure the file is in the root directory.")
except Exception as e:
    print(f"An error occurred: {e}")