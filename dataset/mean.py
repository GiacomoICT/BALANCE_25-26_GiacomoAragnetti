import pandas as pd
import os

def calculate_global_weighted_mean(directory=".\dataset", label_column="label"):
    total_sum = 0.0
    total_count = 0
    
    # Iterate through group0 to group8
    for i in range(9):
        file_name = f"group{i}_combined.csv"
        file_path = os.path.join(directory, file_name)
        print(file_path)
        
        if os.path.exists(file_path):
            try:
                # Using your established semicolon separator
                df = pd.read_csv(file_path, sep=';', on_bad_lines='skip', engine='c')
                
                if label_column in df.columns:
                    # Drop NaNs to ensure the mean is accurate
                    labels = df[label_column].dropna()
                    
                    total_sum += labels.sum()
                    total_count += len(labels)
                    
                    print(f"[File] {file_name}: Count={len(labels)}, Local Mean={labels.mean():.4f}")
                else:
                    print(f"[Warning] Column '{label_column}' not found in {file_name}")
            except Exception as e:
                print(f"[Error] Could not process {file_name}: {e}")
        else:
            print(f"[Warning] {file_name} not found.")

    # Final Weighted Calculation
    if total_count > 0:
        global_mean = total_sum / total_count
        print("-" * 30)
        print(f"Total Entries: {total_count}")
        print(f"Global Weighted Mean: {global_mean:.4f}")
        return global_mean
    else:
        print("No data found to calculate mean.")
        return None

if __name__ == "__main__":
    calculate_global_weighted_mean()