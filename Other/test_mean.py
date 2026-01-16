import pandas as pd
import numpy as np

def generate_constant_label_csv(input_csv_path, output_csv_path, constant_value=76):
    """
    Small script to generate a constant prediction to see what the performance of 
    just the mean at every row was (mean is not best btw, should be median)

    This one as it is a simple check was not adapted to work, so will only work provided 
    the test file is in the same folder and the name of the file is matched in 
    this code 
    """
    try:
        df = pd.read_csv(
            input_csv_path, 
            sep=';', 
            on_bad_lines='skip', 
            engine='c', 
            low_memory=False
        )
        
        num_rows = len(df)
        print(f"[System] Input file '{input_csv_path}' has {num_rows} valid rows.")

        results_df = pd.DataFrame({
            'id': range(num_rows),             # Row indices 0, 1, 2...
            'label': [constant_value] * num_rows  # Every entry is 75
        })

        results_df.to_csv(output_csv_path, index=False)
        print(f"[Success] Constant label CSV saved to '{output_csv_path}'")

    except Exception as e:
        print(f"[Error] Failed to process CSV: {e}")

if __name__ == "__main__":
    generate_constant_label_csv("x_test.csv", "predictions_constant_75.csv")