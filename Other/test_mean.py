import pandas as pd
import numpy as np

def generate_constant_label_csv(input_csv_path, output_csv_path, constant_value=76):
    """
    Reads an input CSV to determine row count and outputs a [id, label] CSV 
    where every label is the constant_value.
    """
    try:
        # 1. Load the input data to get the exact number of rows
        # Using your established settings for semicolon and skipping bad lines
        df = pd.read_csv(
            input_csv_path, 
            sep=';', 
            on_bad_lines='skip', 
            engine='c', 
            low_memory=False
        )
        
        num_rows = len(df)
        print(f"[System] Input file '{input_csv_path}' has {num_rows} valid rows.")

        # 2. Create the results DataFrame in the [id, label] format
        results_df = pd.DataFrame({
            'id': range(num_rows),             # Row indices 0, 1, 2...
            'label': [constant_value] * num_rows  # Every entry is 75
        })

        # 3. Save to CSV without the index column
        results_df.to_csv(output_csv_path, index=False)
        print(f"[Success] Constant label CSV saved to '{output_csv_path}'")

    except Exception as e:
        print(f"[Error] Failed to process CSV: {e}")

if __name__ == "__main__":
    # Example Usage:
    # Change 'x_test.csv' to your actual input filename
    generate_constant_label_csv("x_test.csv", "predictions_constant_75.csv")