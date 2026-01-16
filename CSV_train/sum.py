import pandas as pd
import os
from pathlib import Path

# 1. Get the absolute path of the folder containing THIS script
# If script is in 'CSV_train', script_dir is '.../CSV_train'
script_dir = Path(__file__).resolve().parent

# 2. Define the sibling 'dataset' folder
# Go up to 'father', then down into 'dataset'
target_base_dir = script_dir.parent / "dataset"
target_base_dir.mkdir(parents=True, exist_ok=True)

# 3. Define the group folders location (they are inside the same folder as the script)
# So group0 is at script_dir / "group0"
for group_num in range(9):
    group_name = f"group{group_num}"
    group_path = script_dir / "CSV_train" / group_name  # Changed this line
    
    if not group_path.exists():
        # Debugging print to help you see where it's looking
        print(f"Directory non trovata! Cercavo in: {group_path}")
        continue
    
    csv_files = [f for f in os.listdir(group_path) 
                 if f.endswith('.csv') and f.startswith('dataset_user_')]
    
    if not csv_files:
        print(f"{group_name}: nessun file CSV da unire")
        continue
    
    print(f"\n{group_name}: trovati {len(csv_files)} file CSV")
    
    dfs = []
    for file in csv_files:
        file_path = os.path.join(group_path, file)
        df = pd.read_csv(file_path, sep=';')
        
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
            
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Cleaning Logic
    if 'resp_avgSleepRespirationValue' in combined_df.columns and 'sleep_averageRespirationValue' in combined_df.columns:
        combined_df['resp_avgSleepRespirationValue'] = combined_df['resp_avgSleepRespirationValue'].fillna(
            combined_df['sleep_averageRespirationValue']
        )
        combined_df = combined_df.drop(columns=['sleep_averageRespirationValue'])
    
    columns_to_drop = ['day', 'act_activeTime']
    columns_to_drop = [col for col in columns_to_drop if col in combined_df.columns]
    if columns_to_drop:
        combined_df = combined_df.drop(columns=columns_to_drop)
    
 # --- UPDATED SAVING LOGIC (DIRECTLY IN DATASET) ---
    # target_base_dir is already the "dataset" folder
    
    # Define the output path directly inside 'dataset'
    output_filename = target_base_dir / f"{group_name}_combined.csv"
    
    # Save the file
    combined_df.to_csv(output_filename, index=False, sep=';')
    
    print(f"  → Salvato direttamente in: {output_filename}")
    print(f"  → Dimensioni: {combined_df.shape}")

print("\n✓ Completato!")