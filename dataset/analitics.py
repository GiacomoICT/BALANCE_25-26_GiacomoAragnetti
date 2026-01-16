import pandas as pd
import os

FEATURES = ["act_totalCalories", "ratio_hr_rest_to_avg", "ratio_deep_sleep", "act_intensity_ratio","resp_avgTomorrowSleepRespirationValue","str_avgStressLevel","hr_lastSevenDaysAvgRestingHeartRate","sleep_remSleepSeconds", "sleep_awakeSleepSeconds"]


# --- CONFIGURATION ---
BASE_DIRECTORY = 'dataset'
# Update these to the columns you want to analyze
SELECTED_COLUMNS = FEATURES.append("label")

def analyze_nested_datasets(root_folder, columns):
    if not os.path.exists(root_folder):
        print(f"Error: Folder '{root_folder}' not found.")
        return

    print(f"Scanning directory: {root_folder}\n")

    # os.walk yields (current_path, subdirectories, files)
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                
                try:
                    df = pd.read_csv(file_path,sep=';')
                    
                    # Identify columns that exist in this specific file
                    existing_cols = [c for c in columns if c in df.columns]
                    
                    if not existing_cols:
                        continue

                    print(f"--- FILE: {os.path.relpath(file_path, root_folder)} ---")
                    
                    # Compute stats
                    stats = pd.DataFrame({
                        'Mean': df[existing_cols].mean(),
                        'Variance': df[existing_cols].var(),
                        'Range (Max-Min)': df[existing_cols].max() - df[existing_cols].min()
                    })
                    
                    print(stats)
                    print("-" * 30)
                    
                except Exception as e:
                    print(f"Could not process {file_path}: {e}")

if __name__ == "__main__":
    analyze_nested_datasets(BASE_DIRECTORY, SELECTED_COLUMNS)