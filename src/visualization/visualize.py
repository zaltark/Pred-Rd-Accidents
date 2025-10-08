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
        <title>Exploratory Data Analysis Report</title>
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
            <h1>Exploratory Data Analysis Report</h1>
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
    with open('reports/eda_report.html', 'w') as f:
        f.write(html_content)

# --- Main script ---
output_dir = 'reports/figures'
os.makedirs(output_dir, exist_ok=True)
plots_info = []

try:
    df = pd.read_csv('train.csv')
    df_sample = df.sample(n=1000, random_state=42)

    # --- Visualize Numeric Features (Continuous) ---
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    discrete_cols = ['num_lanes', 'num_reported_accidents', 'speed_limit']
    continuous_cols = [col for col in numeric_cols if col not in discrete_cols and col != 'id']

    for col in continuous_cols:
        # Distribution plot
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        path = os.path.join('figures', f'{col}_distribution.png')
        plt.savefig(os.path.join(output_dir, f'{col}_distribution.png'))
        plt.close()
        plots_info.append({
            'title': f'Distribution of {col}',
            'path': path,
            'explanation': f'This histogram shows the distribution of {col}.'
        })

        # Scatter plot vs. accident_risk
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=df_sample[col], y=df_sample['accident_risk'])
        plt.title(f'{col} vs. Accident Risk')
        plt.xlabel(col)
        plt.ylabel('Accident Risk')
        path = os.path.join('figures', f'{col}_vs_accident_risk.png')
        plt.savefig(os.path.join(output_dir, f'{col}_vs_accident_risk.png'))
        plt.close()
        plots_info.append({
            'title': f'{col} vs. Accident Risk',
            'path': path,
            'explanation': f'This scatter plot shows the relationship between {col} and accident risk. We can look for trends or patterns that suggest a correlation.'
        })


    # --- Visualize Discrete Numeric Features ---
    for col in discrete_cols:
        # Distribution plot
        plt.figure(figsize=(10, 6))
        sns.countplot(x=df[col])
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        path = os.path.join('figures', f'{col}_distribution.png')
        plt.savefig(os.path.join(output_dir, f'{col}_distribution.png'))
        plt.close()
        plots_info.append({
            'title': f'Distribution of {col}',
            'path': path,
            'explanation': f'This bar chart shows the count of each value for the discrete feature {col}.'
        })

        # Box plot vs. accident_risk
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df[col], y=df['accident_risk'], order=sorted(df[col].unique()))
        plt.title(f'{col} vs. Accident Risk')
        plt.xlabel(col)
        plt.ylabel('Accident Risk')
        path = os.path.join('figures', f'{col}_vs_accident_risk.png')
        plt.savefig(os.path.join(output_dir, f'{col}_vs_accident_risk.png'))
        plt.close()
        plots_info.append({
            'title': f'{col} vs. Accident Risk',
            'path': path,
            'explanation': f'This box plot shows the distribution of accident risk for each value of {col}. We can see if there are significant differences in risk for different values of this feature.'
        })


    # --- Visualize Categorical Features ---
    categorical_cols = df.select_dtypes(include=['object', 'bool']).columns
    for col in categorical_cols:
        # Distribution plot
        plt.figure(figsize=(10, 6))
        sns.countplot(y=df[col], order = df[col].value_counts().index)
        plt.title(f'Distribution of {col}')
        plt.xlabel('Count')
        plt.ylabel(col)
        plt.tight_layout()
        path = os.path.join('figures', f'{col}_distribution.png')
        plt.savefig(os.path.join(output_dir, f'{col}_distribution.png'))
        plt.close()
        plots_info.append({
            'title': f'Distribution of {col}',
            'path': path,
            'explanation': f'This bar chart shows the frequency of each category for the feature {col}.'
        })

        # Box plot vs. accident_risk
        plt.figure(figsize=(10, 6))
        if df[col].dtype == 'bool':
            sns.boxplot(y=df[col].astype(str), x=df['accident_risk'])
        else:
            sns.boxplot(y=df[col], x=df['accident_risk'], order = df[col].value_counts().index)
        plt.title(f'{col} vs. Accident Risk')
        plt.ylabel(col)
        plt.xlabel('Accident Risk')
        plt.tight_layout()
        path = os.path.join('figures', f'{col}_vs_accident_risk.png')
        plt.savefig(os.path.join(output_dir, f'{col}_vs_accident_risk.png'))
        plt.close()
        plots_info.append({
            'title': f'{col} vs. Accident Risk',
            'path': path,
            'explanation': f'This box plot shows the distribution of accident risk for each category of {col}. We can look for differences in the median and spread of risk across categories.'
        })

    # --- Generate HTML Report ---
    generate_html_report(plots_info)
    print(f"Visualizations and HTML report saved to {output_dir} and reports/eda_report.html")

except FileNotFoundError:
    print("Error: train.csv not found. Make sure the file is in the root directory.")
except Exception as e:
    print(f"An error occurred: {e}")
