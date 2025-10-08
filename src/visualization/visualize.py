import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Create the output directory if it doesn't exist
output_dir = 'reports/figures'
os.makedirs(output_dir, exist_ok=True)

# Load the dataset
try:
    df = pd.read_csv('train.csv')

    # --- Visualize Numeric Features (Continuous) ---
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    discrete_cols = ['num_lanes', 'num_reported_accidents', 'speed_limit']
    continuous_cols = [col for col in numeric_cols if col not in discrete_cols and col != 'id']

    for col in continuous_cols:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(output_dir, f'{col}_distribution.png'))
        plt.close()

    # --- Visualize Discrete Numeric Features ---
    for col in discrete_cols:
        plt.figure(figsize=(10, 6))
        sns.countplot(x=df[col])
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.savefig(os.path.join(output_dir, f'{col}_distribution.png'))
        plt.close()


    # --- Visualize Categorical Features ---
    categorical_cols = df.select_dtypes(include=['object', 'bool']).columns
    for col in categorical_cols:
        plt.figure(figsize=(10, 6))
        sns.countplot(y=df[col], order = df[col].value_counts().index)
        plt.title(f'Distribution of {col}')
        plt.xlabel('Count')
        plt.ylabel(col)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{col}_distribution.png'))
        plt.close()

    print(f"Visualizations saved to {output_dir}")

except FileNotFoundError:
    print("Error: train.csv not found. Make sure the file is in the root directory.")
except Exception as e:
    print(f"An error occurred: {e}")