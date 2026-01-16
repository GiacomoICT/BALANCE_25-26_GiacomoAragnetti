import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestRegressor

OUTPUT_DIR = 'dataset/results'

def extract_all_groups_local(target_col='label'):
    all_importances = []
    feature_names = None
    processed_files = []

    # Iterate through Groups 0 to 8
    for i in range(9):
        file_name = f"dataset/group{i}_combined_v2.csv"
        # Since it's in the same folder, we just use the name
        if not os.path.exists(file_name):
            print(f"[-] Skipping: {file_name} not found in current directory.")
            continue
            
        print(f"[+] Processing {file_name}...")
        df = pd.read_csv(file_name, sep=';')
        
        # 1. Clean data: Drop non-numeric columns and target
        # We also drop the 'label' (target) and any ID columns if they exist
        X = df.drop(columns=[f for f in target_col if f in df.columns])
        X = X.select_dtypes(include=[np.number]) # Only keep numeric features
        X = X.fillna(0) # Random Forest can't handle NaNs
        
        y = df['label'].values
        
        if feature_names is None:
            feature_names = X.columns
            
        # 2. Train Random Forest (using 100 trees for speed/accuracy balance)
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        all_importances.append(rf.feature_importances_)
        processed_files.append(file_name)

    if not all_importances:
        print("[!] Error: No 'groupX_combined.csv' files found in the current folder.")
        return

    # 3. Aggregate results
    avg_importance = np.mean(all_importances, axis=0)
    std_importance = np.std(all_importances, axis=0)

    # 4. Create and Save Rankings
    results = pd.DataFrame({
        'Feature': feature_names,
        'Global_Mean_Importance': avg_importance,
        'Inter_Node_StdDev': std_importance
    }).sort_values(by='Global_Mean_Importance', ascending=False)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    results.to_csv('dataset/results/aggregated_feature_importance.csv', index=False)
    
    # 5. Visualizing the Top 20 Features
    plt.figure(figsize=(12, 10))
    top_n = results.head(20)
    plt.barh(top_n['Feature'], top_n['Global_Mean_Importance'], xerr=top_n['Inter_Node_StdDev'], color='teal')
    plt.xlabel('Mean Importance')
    plt.title(f'Top 20 Features (Averaged over {len(processed_files)} Groups)')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('feature_importance_ranking.png')
    
    # 6. Summary Output
    print("\n" + "="*30)
    print("FEATURE EXTRACTION SUMMARY")
    print("="*30)
    print(f"Nodes processed: {len(processed_files)}")
    print(f"Top Feature: {results.iloc[0]['Feature']} ({results.iloc[0]['Global_Mean_Importance']:.4f})")
    
    # Identify useless features
    useless = results[results['Global_Mean_Importance'] < 0.001]
    print(f"Features with < 0.1% importance: {len(useless)}")
    if len(useless) > 0:
        print(f"Suggested to drop: {list(useless['Feature'].head(5))}...")

    return results

if __name__ == "__main__":
    # Run the script
    # This assumes your CSVs use 'label' as the target. Adjust if needed.
    importance_df = extract_all_groups_local(target_col=["hr_time_series", "resp_time_series", "stress_time_series", "label"])