import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def generate_deep_dive_html(plot_path, explanation):
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
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Deep Dive: The Effect of Road Signs on Accident Risk</h1>
            <div class="plot">
                <h2>Road Signs vs. Accident Risk, Faceted by Road Type</h2>
                <img src="{plot_path}" alt="Faceted Box Plot of Road Signs vs. Accident Risk">
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

    # Create the faceted plot
    g = sns.catplot(
        data=df,
        x='road_signs_present',
        y='accident_risk',
        col='road_type',
        kind='box',
        height=6,
        aspect=0.8
    )
    g.fig.suptitle('Effect of Road Signs on Accident Risk, by Road Type', y=1.03)
    g.set_axis_labels("Road Signs Present", "Accident Risk")
    plot_path_on_disk = os.path.join(output_dir, 'road_signs_faceted_by_road_type.png')
    plt.savefig(plot_path_on_disk)
    plt.close()

    # Generate the HTML report
    plot_path_for_html = os.path.join('figures', 'road_signs_faceted_by_road_type.png')
    explanation = """
    <p>This plot investigates the effect of road signs on accident risk, while controlling for the type of road. We do this because the effect of a road sign might be different depending on the context of the road (urban, rural, or highway).</p>
    <p><strong>How to read this plot:</strong> Each of the three plots above shows a comparison of accident risk for roads with and without signs, but only for a specific road type.</p>
    <ul>
        <li><strong>Urban:</strong> Compare the 'True' and 'False' boxes in the 'urban' plot.</li>
        <li><strong>Rural:</strong> Compare the 'True' and 'False' boxes in the 'rural' plot.</li>
        <li><strong>Highway:</strong> Compare the 'True' and 'False' boxes in the 'highway' plot.</li>
    </ul>
    <p><strong>What to look for:</strong> If, within a specific road type, the median accident risk (the line in the middle of the box) is lower for 'True' (signs present) than for 'False' (signs not present), it suggests that road signs have a positive safety effect in that context. If the boxes are at a similar level, it suggests that road signs may not have a significant effect for that road type, or that other confounding variables are still at play.</p>
    """
    generate_deep_dive_html(plot_path_for_html, explanation)

    print(f"Deep dive visualization and HTML report created successfully.")

except FileNotFoundError:
    print("Error: train.csv not found. Make sure the file is in the root directory.")
except Exception as e:
    print(f"An error occurred: {e}")
