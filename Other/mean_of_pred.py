import pandas as pd
import os

def compute_label_mean(file_path):
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        return

    try:
        # Load the CSV
        df = pd.read_csv(file_path)

        # Check if the column exists
        if 'label' in df.columns:
            # Compute the mean
            label_mean = df['label'].mean()
            label_var = df['label'].var()
            print(f"The mean of the 'label' column is: {label_mean:.4f}")
            print(f"Var of pred col: {label_var:.4f}")
        else:
            print("Error: The column 'label' does not exist in this CSV.")
            print(f"Available columns: {list(df.columns)}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Usage
csv_filename = 'predictions.csv'  # Replace with your actual file path
compute_label_mean(csv_filename)