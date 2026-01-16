import pandas as pd
import numpy as np
import glob
import os
from collections import Counter
import ast

INPUT_PATTERN = 'dataset/group*_combined.csv'
OUTPUT_DIR = 'dataset/results'
TS_COLUMNS = ['hr_time_series', 'resp_time_series', 'stress_time_series']

def get_list_length(val):
    if pd.isna(val) or str(val).strip() in ["", "[]"]:
        return 0
    try:
        if isinstance(val, str):
            parsed = ast.literal_eval(val)
            return len(parsed)
        return len(val)
    except:
        return -1

def analyze_lengths():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    files = glob.glob(INPUT_PATTERN)
    
    if not files:
        print(f"No files found matching {INPUT_PATTERN}")
        return

    print(f"Analyzing {len(files)} files...")

    for filepath in files:
        base_name = os.path.basename(filepath)
        group_id = base_name.split('_')[0] 
        report_path = os.path.join(OUTPUT_DIR, f"{group_id}_ts_stats.txt")
        
        print(f"Processing {base_name}...")
        
        try:
            df = pd.read_csv(filepath, sep=';', on_bad_lines='skip', usecols=lambda x: x in TS_COLUMNS)
            
            with open(report_path, 'w') as f:
                f.write(f"--- Time Series Length Report: {group_id} ---\n")
                f.write(f"Source file: {filepath}\n")
                f.write("=" * 40 + "\n\n")
                
                for col in TS_COLUMNS:
                    if col in df.columns:
                        # Calculate lengths
                        lengths = df[col].apply(get_list_length)
                        stats = Counter(lengths)
                        
                        f.write(f"COLUMN: {col}\n")
                        f.write("-" * 20 + "\n")
                        # Sort by length for a readable histogram
                        for length in sorted(stats.keys()):
                            count = stats[length]
                            f.write(f"Length: {length:5} | Count: {count}\n")
                        f.write("\n")
                    else:
                        f.write(f"COLUMN: {col} -> NOT FOUND\n\n")
            
            print(f"  -> Report saved to: {report_path}")

        except Exception as e:
            print(f"  Error processing {filepath}: {e}")

if __name__ == "__main__":
    analyze_lengths()