import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path


"""
Small script to calculate correlation between label and features in the datasets
Was used as part of early feature extraction, results showed little promise so 
it was later abandoned in favor of more robust feature extraction based on random forests
"""


script_dir = Path(__file__).resolve().parent
dataset_path = script_dir.parent / "dataset" # Adjust if your folder name is different
files = glob.glob(os.path.join(dataset_path, 'group*_combined.csv'))


results_dir = script_dir.parent / "results"
results_dir.mkdir(parents=True, exist_ok=True)
output_file = results_dir / "correlation_results.txt"

print(f"--- Found {len(files)} group files ---")

# This list will hold the correlation series from each group
correlation_results = []

for file_path in sorted(files):
    try:

        df = pd.read_csv(file_path, sep=';', on_bad_lines='skip', low_memory=False)
        numeric_df = df.select_dtypes(include=[np.number])
        
        if 'label' not in numeric_df.columns:
            print(f"Skipping {file_path}: 'label' column not found.")
            continue
            
        # Calculate absolute correlation with the label
        corr_series = numeric_df.corr()['label'].abs()
        
        corr_series.name = os.path.basename(file_path)
        
        correlation_results.append(corr_series)
        print(f"Processed: {file_path}")
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

all_corrs_df = pd.concat(correlation_results, axis=1)

# Calculate the average across all groups
average_correlations = all_corrs_df.mean(axis=1).sort_values(ascending=False)

print(f"\nSaving results to {output_file}...")

with open(output_file, "w", encoding="utf-8") as f:
    f.write("--- TOP UNIVERSAL FEATURES (Averaged across all nodes) ---\n")
    f.write(average_correlations.to_string())

print("✓ Results saved successfully!")