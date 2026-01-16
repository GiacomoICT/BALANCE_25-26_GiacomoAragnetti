import pandas as pd
import numpy as np
import glob
import os

# 1. Setup paths
dataset_path = 'dataset'  # Adjust if your folder name is different
files = glob.glob(os.path.join(dataset_path, 'group*_combined.csv'))

print(f"--- Found {len(files)} group files ---")

# 2. This list will hold the correlation series from each group
correlation_results = []

for file_path in sorted(files):
    try:
        # Load the group data
        df = pd.read_csv(file_path, sep=';', on_bad_lines='skip', low_memory=False)
        
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if 'label' not in numeric_df.columns:
            print(f"Skipping {file_path}: 'label' column not found.")
            continue
            
        # Calculate absolute correlation with the label
        # This creates a Pandas Series
        corr_series = numeric_df.corr()['label'].abs()
        
        # Give the series a name (the filename) to keep track
        corr_series.name = os.path.basename(file_path)
        
        correlation_results.append(corr_series)
        print(f"Processed: {file_path}")
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# 3. Combine all results into one DataFrame
# Each column will be a different group's correlation scores
all_corrs_df = pd.concat(correlation_results, axis=1)

# 4. Calculate the average across all groups
# We ignore NaNs in case some groups are missing certain columns
average_correlations = all_corrs_df.mean(axis=1).sort_values(ascending=False)

# 5. Display the final Top 15 "Universal" features
print("\n--- TOP UNIVERSAL FEATURES (Averaged across all nodes) ---")
print(average_correlations.head(16))