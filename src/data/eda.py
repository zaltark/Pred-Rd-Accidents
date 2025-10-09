import pandas as pd

# Load the dataset
try:
    df = pd.read_csv('train.csv')

    # --- Initial EDA ---
    print("--- Head of the dataset ---")
    print(df.head())
    print("\n" + "="*50 + "\n")

    print("--- Dataset Info ---")
    # Redirecting info() output to a string to print it
    info_str = df.info(buf=None)
    print(info_str)
    print("\n" + "="*50 + "\n")

    print("--- Descriptive Statistics ---")
    print(df.describe())
    print("\n" + "="*50 + "\n")

    print("--- Missing Values ---")
    print(df.isnull().sum())
    print("\n" + "="*50 + "\n")

except FileNotFoundError:
    print("Error: train.csv not found. Make sure the file is in the root directory.")
except Exception as e:
    print(f"An error occurred: {e}")
